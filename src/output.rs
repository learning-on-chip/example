use sqlite::{Connection, Statement, State};
use std::mem;

use streamer::{Config, Result};
use streamer::platform::{self, Platform, Profile};
use streamer::system::{Event, EventKind};

pub struct Output {
    connection: Connection,
    markers: Statement<'static>,
    profiles: Statement<'static>,
    processed: usize,
    written: usize,
    reduction: usize,
    buffers: (Vec<f64>, Vec<f64>),
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
        let reduction = *config.get::<i64>("reduction").unwrap_or(&1);
        if reduction <= 0 {
            raise!("the reduction size should be greater than zero");
        }
        Ok(Output {
            connection: connection,
            markers: markers,
            profiles: profiles,
            processed: 0,
            written: 0,
            reduction: reduction as usize,
            buffers: (vec![0.0; units], vec![0.0; units]),
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
        let &mut Output { markers: ref mut statement, written, .. } = self;
        let (kind, mapping) = match &event.kind {
            &EventKind::Start(_, ref mapping) => (0, mapping),
            &EventKind::Finish(_, ref mapping) => (1, mapping),
            _ => return Ok(()),
        };
        for &(_, j) in mapping {
            ok!(statement.reset());
            ok!(statement.bind(1, written as i64));
            ok!(statement.bind(2, j as i64));
            ok!(statement.bind(3, kind as i64));
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
        }
        Ok(())
    }

    fn write_profiles(&mut self, profiles: &(Profile, Profile)) -> Result<()> {
        let &Profile { units, steps, data: ref new_power, .. } = &profiles.0;
        let &Profile { data: ref new_temperature, .. } = &profiles.1;
        let &mut Output {
            profiles: ref mut statement, ref mut processed, ref mut written,
            reduction, buffers: (ref mut power, ref mut temperature), ..
        } = self;
        for i in 0..steps {
            for j in 0..units {
                power[j] += new_power[i * units + j];
                temperature[j] += new_temperature[i * units + j];
            }
            *processed += 1;
            if *processed % reduction > 0 {
                continue;
            }
            ok!(statement.reset());
            let mut k = 0;
            for j in 0..units {
                power[j] /= reduction as f64;
                temperature[j] /= reduction as f64;
                ok!(statement.bind(k + 1, *written as i64));
                ok!(statement.bind(k + 2, j as i64));
                ok!(statement.bind(k + 3, power[j]));
                ok!(statement.bind(k + 4, temperature[j]));
                power[j] = 0.0;
                temperature[j] = 0.0;
                k += 4;
            }
            if State::Done != ok!(statement.next()) {
                raise!("failed to write into the database");
            }
            *written += 1;
        }
        Ok(())
    }
}
