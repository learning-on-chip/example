use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{self, Platform, Profile};
use streamer::system::{Event, EventKind};

pub struct Output {
    connection: Connection,
    markers: Statement<'static>,
    profiles: Statement<'static>,
    position: usize,
}

impl Output {
    pub fn new(platform: &platform::Thermal, config: &Config) -> Result<Self> {
        use sql::prelude::*;

        let units = platform.elements().len();
        let connection = ok!(Connection::open(path!(@unchecked config,
                                                    "an output file is required")));
        ok!(connection.execute("
            PRAGMA journal_mode = MEMORY;
            PRAGMA synchronous = OFF;
        "));
        let markers = {
            ok!(connection.execute(
                ok!(create_table("markers").if_not_exists().columns(&[
                    "sequence_id".integer().not_null(),
                    "component_id".integer().not_null(),
                    "kind".integer().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("markers").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("markers").columns(&[
                    "sequence_id", "component_id", "kind",
                ]).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        let profiles = {
            ok!(connection.execute(
                ok!(create_table("profiles").if_not_exists().columns(&[
                    "sequence_id".integer().not_null(),
                    "component_id".integer().not_null(),
                    "power".float().not_null(),
                    "temperature".float().not_null(),
                ]).compile())
            ));
            ok!(connection.execute(ok!(delete_from("profiles").compile())));
            let statement = ok!(connection.prepare(
                ok!(insert_into("profiles").columns(&[
                    "sequence_id", "component_id", "power", "temperature",
                ]).batch(units).compile())
            ));
            unsafe { mem::transmute(statement) }
        };
        Ok(Output {
            connection: connection,
            markers: markers,
            profiles: profiles,
            position: 0,
        })
    }

    pub fn next(&mut self, event: &Event, profiles: &(Profile, Profile)) -> Result<()> {
        ok!(self.connection.execute("BEGIN TRANSACTION"));
        ok!(self.write_profiles(profiles));
        ok!(self.write_markers(event));
        ok!(self.connection.execute("END TRANSACTION"));
        Ok(())
    }

    fn write_markers(&mut self, event: &Event) -> Result<()> {
        let &mut Output { markers: ref mut statement, position, .. } = self;
        let (kind, mapping) = match &event.kind {
            &EventKind::Start(_, ref mapping) => (0, mapping),
            &EventKind::Finish(_, ref mapping) => (1, mapping),
            _ => return Ok(()),
        };
        for &(_, j) in mapping {
            ok!(statement.reset());
            ok!(statement.bind(1, position as i64));
            ok!(statement.bind(2, j as i64));
            ok!(statement.bind(3, kind as i64));
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
        }
        Ok(())
    }

    fn write_profiles(&mut self, profiles: &(Profile, Profile)) -> Result<()> {
        let &Profile { units, steps, data: ref power, .. } = &profiles.0;
        let &Profile { data: ref temperature, .. } = &profiles.1;
        let &mut Output { profiles: ref mut statement, ref mut position, .. } = self;
        for i in 0..steps {
            ok!(statement.reset());
            for j in 0..units {
                ok!(statement.bind(4 * j + 1, *position as i64));
                ok!(statement.bind(4 * j + 2, j as i64));
                ok!(statement.bind(4 * j + 3, power[i * units + j]));
                ok!(statement.bind(4 * j + 4, temperature[i * units + j]));
            }
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
            *position += 1;
        }
        Ok(())
    }
}
